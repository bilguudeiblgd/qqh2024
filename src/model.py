import numpy as np
import pandas as pd
from heapq import nlargest
from collections import defaultdict
from collections import deque
from xgboost import XGBClassifier

class PredictionModel:
    def __init__(self):
        self.train_data = pd.DataFrame()  # Store historical data for training
        self.model = XGBClassifier(eval_metric="logloss")  # Gradient Boosting Model

    def add_game_result(self, game_features, result):
        """
        Add a game result to the training data.
        :param game_features: Dictionary of features for the game.
        :param result: Binary result (1 for win, 0 for loss).
        """
        game_features["result"] = result
        self.train_data = pd.concat([self.train_data, pd.DataFrame([game_features])], ignore_index=True)

    def train_model(self):
        """
        Train the model on the current training data.
        """
        if self.train_data.empty:
            print("No data to train on.")
            return
        
        X = self.train_data.drop(columns=["result"])
        y = self.train_data["result"]
        self.model.fit(X, y)

    def predict_outcome(self, game_features):
        """
        Predict the probability of winning for the given game.
        :param game_features: Dictionary of features for the game.
        :return: Probability of winning.
        """
        features_df = pd.DataFrame([game_features])
        return self.model.predict_proba(features_df)[:, 1]  # Probability of class 1 (win)

class Model:
    iteration = 0
    def __init__(self):
        # Initialize
        self.counter = 0
        self.team_map = defaultdict(lambda: {
            "home_games": deque(),
            "away_games": deque(),
            "home_scored": deque(),
            "home_allowed": deque(),
            "away_scored": deque(),
            "away_allowed": deque(),
        })
        # Player-level stats
        self.player_map = defaultdict(lambda: {
            "team_id": None,
            "games_played": 0,
            "total_stats": defaultdict(float),  # Tracks cumulative stats
        })
        
        # Team-to-players mapping
        self.team_players = defaultdict(set)

        self.leaderboards = {
            "assists": [],
            "blocks_steals": [],
            "points": [],
            "true_shooting": [],
        }

        self.prediction_model = PredictionModel()


    def update_leaderboards(self, players_df: pd.DataFrame):
        """
        Update the leaderboards for top players in assists, blocks + steals, points, and true shooting.
        :param players_df: DataFrame containing player statistics.
        """
        # Create a copy of the DataFrame for safe modifications
        players_df = players_df.copy()

        # True Shooting Percentage calculation
        players_df["TS%"] = 2 * (players_df["PTS"] / (2 * (players_df["FGA"] + 0.44 * players_df["FTA"])))

        # Assists leaderboard
        assists = players_df.groupby("Player")["AST"].sum().sort_values(ascending=False).head(10)
        self.leaderboards["assists"] = assists.index.tolist()

        # Blocks + steals leaderboard
        defense = (players_df.groupby("Player")["BLK"].sum() + players_df.groupby("Player")["STL"].sum())
        defense = defense.sort_values(ascending=False).head(10)
        self.leaderboards["blocks_steals"] = defense.index.tolist()

        # Points leaderboard
        points = players_df.groupby("Player")["PTS"].sum().sort_values(ascending=False).head(10)
        self.leaderboards["points"] = points.index.tolist()

        # True shooting percentage leaderboard
        true_shooting = players_df.groupby("Player")["TS%"].mean().sort_values(ascending=False).head(10)
        self.leaderboards["true_shooting"] = true_shooting.index.tolist()



    def get_leaderboard(self, category: str):
        """
        Retrieve the top 10 players for a given category.
        :param category: The category to retrieve (e.g., 'assists', 'blocks_steals', 'points', 'true_shooting').
        :return: List of player IDs and their values for the category.
        """
        if category not in self.leaderboards:
            raise ValueError(f"Invalid category: {category}")

        leaderboard = []
        for player_id in self.leaderboards[category]:
            if category == "assists":
                value = self.player_map[player_id]["total_stats"]["AST"]
            elif category == "blocks_steals":
                value = self.player_map[player_id]["total_stats"]["BLK"] + self.player_map[player_id]["total_stats"]["STL"]
            elif category == "points":
                value = self.player_map[player_id]["total_stats"]["PTS"]
            elif category == "true_shooting":
                value = self.calculate_true_shooting_percentage(player_id)
            leaderboard.append((player_id, value))

        return leaderboard


    def update_team_map(self, games: pd.DataFrame):
        """
        Update team_map with the latest game data, calculating aggregate team statistics.
        :param games: DataFrame containing game statistics.
        """
        for _, game in games.iterrows():
            home_team = game["HID"]
            away_team = game["AID"]

            # Ensure team_map structure exists for both teams
            if home_team not in self.team_map:
                self.team_map[home_team] = {
                    "home_games": deque(maxlen=10),
                    "away_games": deque(maxlen=10),
                    "stats": defaultdict(float)  # Store cumulative stats for the team
                }
            if away_team not in self.team_map:
                self.team_map[away_team] = {
                    "home_games": deque(maxlen=10),
                    "away_games": deque(maxlen=10),
                    "stats": defaultdict(float)
                }

            # Add game data to home and away teams
            self.team_map[home_team]["home_games"].append(game.to_dict())
            self.team_map[away_team]["away_games"].append(game.to_dict())

            # Update team stats for the home team
            home_stats = self.team_map[home_team]["stats"]
            home_stats["points_scored"] += game["HSC"]
            home_stats["points_allowed"] += game["ASC"]
            home_stats["field_goals_made"] += game["HFGM"]
            home_stats["field_goal_attempts"] += game["HFGA"]
            home_stats["free_throws_made"] += game["HFTM"]
            home_stats["free_throw_attempts"] += game["HFTA"]
            home_stats["three_pointers_made"] += game["HFG3M"]
            home_stats["three_pointer_attempts"] += game["HFG3A"]
            home_stats["offensive_rebounds"] += game["HORB"]
            home_stats["defensive_rebounds"] += game["HDRB"]
            home_stats["total_rebounds"] += game["HRB"]
            home_stats["assists"] += game["HAST"]
            home_stats["turnovers"] += game["HTOV"]
            home_stats["blocks"] += game["HBLK"]
            home_stats["steals"] += game["HSTL"]
            home_stats["personal_fouls"] += game["HPF"]
            home_stats["opponent_offensive_rebounds"] += game["AORB"]
            home_stats["games_played"] += 1

            # Update team stats for the away team
            away_stats = self.team_map[away_team]["stats"]
            away_stats["points_scored"] += game["ASC"]
            away_stats["points_allowed"] += game["HSC"]
            away_stats["field_goals_made"] += game["AFGM"]
            away_stats["field_goal_attempts"] += game["AFGA"]
            away_stats["free_throws_made"] += game["AFTM"]
            away_stats["free_throw_attempts"] += game["AFTA"]
            away_stats["three_pointers_made"] += game["AFG3M"]
            away_stats["three_pointer_attempts"] += game["AFG3A"]
            away_stats["offensive_rebounds"] += game["AORB"]
            away_stats["defensive_rebounds"] += game["ADRB"]
            away_stats["total_rebounds"] += game["ARB"]
            away_stats["assists"] += game["AAST"]
            away_stats["turnovers"] += game["ATOV"]
            away_stats["blocks"] += game["ABLK"]
            away_stats["steals"] += game["ASTL"]
            away_stats["personal_fouls"] += game["APF"]
            away_stats["opponent_offensive_rebounds"] += game["HORB"]
            away_stats["games_played"] += 1

    def get_team_averages(self, team_id: int):
        """
        Retrieve the averages of scored and allowed points for the given team.
        :param team_id: Team identifier.
        :return: A dictionary with averages for the last 10 home and away games.
        """
        team_data = self.team_map[team_id]
        return {
            "avg_home_scored": sum(team_data["home_scored"]) / len(team_data["home_scored"]) if team_data["home_scored"] else 0,
            "avg_home_allowed": sum(team_data["home_allowed"]) / len(team_data["home_allowed"]) if team_data["home_allowed"] else 0,
            "avg_away_scored": sum(team_data["away_scored"]) / len(team_data["away_scored"]) if team_data["away_scored"] else 0,
            "avg_away_allowed": sum(team_data["away_allowed"]) / len(team_data["away_allowed"]) if team_data["away_allowed"] else 0,
        }
    
    def update_player_map(self, players: pd.DataFrame):
        """Update player statistics and associate players with teams."""
        for _, player in players.iterrows():
            player_id = player["Player"]
            team_id = player["Team"]

            # Add player to team
            self.team_players[team_id].add(player_id)

            # Update player's team association
            if self.player_map[player_id]["team_id"] is None:
                self.player_map[player_id]["team_id"] = team_id

            # Update cumulative stats
            stats = self.player_map[player_id]["total_stats"]
            for stat in ["MIN", "FGM", "FGA", "FG3M", "FG3A", "FTM", "FTA", "ORB", "DRB", "RB", "AST", "STL", "BLK", "TOV", "PF", "PTS"]:
                stats[stat] += player[stat]

            # Increment games played
            self.player_map[player_id]["games_played"] += 1

    def get_player_averages(self, player_id: int):
        """
        Retrieve a player's average statistics across all games played.
        :param player_id: Player identifier.
        :return: A dictionary with average stats.
        """
        player_data = self.player_map[player_id]
        games_played = player_data["games_played"]

        if games_played == 0:
            return {}

        averages = {stat: total / games_played for stat, total in player_data["total_stats"].items()}
        return averages

    def get_team_players(self, team_id: int):
        """
        Retrieve the IDs of players associated with a given team.
        :param team_id: Team identifier.
        :return: A set of player IDs.
        """
        return self.team_players[team_id]

    def calculate_uPER(
        self,
        min_played, three_point_made, player_pf, lg_ft, lg_pf, player_ft, 
        tm_ast, tm_fg, player_fg, assist_factor, player_ast, vop, drbp, 
        player_orb, player_blk, player_fta, player_fga, player_trb, 
        lg_fta, player_to, player_stl
    ) -> float:
        """
        Calculate unadjusted Player Efficiency Rating (uPER).
        """
        component_1 = three_point_made
        component_2 = (player_pf * lg_ft) / lg_pf
        component_3 = (player_ft / 2) * (2 - (tm_ast / (3 * tm_fg)))
        component_4 = player_fg * (2 - (assist_factor * tm_ast / tm_fg))
        component_5 = (2 * player_ast) / 3
        component_6 = vop * (
            drbp * (2 * player_orb + player_blk - 0.2464 * (player_fta - player_ft) - (player_fga - player_fg) - player_trb)
            + (0.44 * lg_fta * player_pf) / lg_pf
            - (player_to + player_orb)
            + player_stl
            + player_trb
            - 0.1936 * (player_fta - player_ft)
        )
        
        uPER = (1 / min_played) * (component_1 - component_2 + component_3 + component_4 + component_5 + component_6)
        return max(0, uPER)

    def calculate_uPER_for_player(self, player_id: int, league_stats: dict, team_stats: dict) -> float:
        """
        Calculate uPER for a single player using cumulative stats and league-wide values.
        :param player_id: Player identifier.
        :param league_stats: Dictionary containing league-wide stats (e.g., lg_ft, lg_pf, lg_fta).
        :param team_stats: Dictionary containing team-specific stats (e.g., tm_ast, tm_fg, assist_factor, vop, drbp).
        :return: Calculated uPER for the player.
        """
        player_data = self.player_map[player_id]["total_stats"]

        # Extract player-specific stats
        min_played = player_data["MIN"]
        if min_played == 0:  # Avoid division by zero
            return 0.0
        three_point_made = player_data["FG3M"]
        player_pf = player_data["PF"]
        player_ft = player_data["FTM"]
        player_fg = player_data["FGM"]
        player_ast = player_data["AST"]
        player_orb = player_data["ORB"]
        player_blk = player_data["BLK"]
        player_fta = player_data["FTA"]
        player_fga = player_data["FGA"]
        player_trb = player_data["RB"]
        player_to = player_data["TOV"]
        player_stl = player_data["STL"]

        # Extract league and team stats

        total_number_games_team = team_stats["games_played"]
        lg_ft = league_stats["lg_ft"]
        lg_pf = league_stats["lg_pf"]
        lg_fta = league_stats["lg_fta"]
        tm_ast = team_stats["assists"]/ total_number_games_team
        tm_fg = team_stats["field_goal_attempts"] /total_number_games_team
        assist_factor = team_stats["assist_factor"]/total_number_games_team
        AVG_N_POSESSIONS = 50
        vop =  ((team_stats["points_scored"]/ total_number_games_team) / AVG_N_POSESSIONS)
        drbp = team_stats["defensive_rebounds"] / (team_stats["defensive_rebounds"]
                                                    + team_stats["opponent_offensive_rebounds"])


        # Calculate and return uPER
        return self.calculate_uPER(
            min_played, three_point_made, player_pf, lg_ft, lg_pf, player_ft, 
            tm_ast, tm_fg, player_fg, assist_factor, player_ast, vop, drbp, 
            player_orb, player_blk, player_fta, player_fga, player_trb, 
            lg_fta, player_to, player_stl
        )
    
    def calculate_team_uPER(self, team_id: int, league_stats: dict, team_stats: dict):
        total_uPER = 0.0
        for player_id in self.get_team_players(team_id):
            total_uPER += self.calculate_uPER_for_player(player_id, league_stats, team_stats)
        return total_uPER

    def calculate_true_shooting_percentage(self, player_id: int):
        """
        Calculate the True Shooting Percentage (TS%) for a given player.
        :param player_id: Player identifier.
        :return: True Shooting Percentage as a float.
        """
        player_data = self.player_map[player_id]
        stats = player_data["total_stats"]

        points = stats["PTS"]
        fga = stats["FGA"]
        fta = stats["FTA"]

        if fga == 0 and fta == 0:
            return 0.0  # Avoid division by zero

        ts_percentage = (points / (2 * (fga + 0.44 * fta))) * 100
        return ts_percentage

    def update_averages(self):
        # Calculate the average points scored and allowed for each team
        for team_id, (home_games, away_games) in self.team_map.items():
            # Combine all games (home and away)
            all_games = home_games + away_games
            
            # Calculate the average points scored and allowed if there are games
            if all_games:
                avg_scored = sum(game[0] for game in all_games) / len(all_games)
                avg_allowed = sum(game[1] for game in all_games) / len(all_games)
            else:
                avg_scored = avg_allowed = 0  # Default to 0 if no games

            # Update the average points map
            self.average_points_map[team_id] = (avg_scored, avg_allowed)

    def calculate_opponent_orb(self, team_id: int):
        """
        Calculate the total offensive rebounds for the opponent teams in the last 10 games.
        :param team_id: The team identifier.
        :return: Total offensive rebounds for opponents.
        """
        opponent_orb = 0

        # Retrieve last 10 home and away games from team_map
        for game in self.team_map[team_id]["home_games"]:
            opponent_orb += game.get("A_ORB", 0)  # Away team's offensive rebounds in home games

        for game in self.team_map[team_id]["away_games"]:
            opponent_orb += game.get("H_ORB", 0)  # Home team's offensive rebounds in away games

        return opponent_orb

    def calculate_features_for_game(self, game: pd.Series, is_home_team: bool, league_stats: dict, team_stats: dict):
        team_id = game["HID"] if is_home_team else game["AID"]
        team_features = {
            "team_uPer": self.calculate_team_uPER(team_id, league_stats, team_stats),
            # "num_top_players_in_assists": sum(1 for pid in self.get_team_players(team_id) if pid in self.leaderboards["assists"]),
            # "num_top_players_in_defense": sum(1 for pid in self.get_team_players(team_id) if pid in self.leaderboards["blocks_steals"]),
            # "num_top_players_in_true_shooting": sum(1 for pid in self.get_team_players(team_id) if pid in self.leaderboards["true_shooting"]),
            # "num_top_players_in_scoring": sum(1 for pid in self.get_team_players(team_id) if pid in self.leaderboards["points"]),
            # "avg_scored_points_last_10": sum(self.team_map[team_id]["home_scored"] + self.team_map[team_id]["away_scored"]) / min(self.team_map[team_id]["games_played"], 10),
            # "avg_allowed_points_last_10": sum(self.team_map[team_id]["home_allowed"] + self.team_map[team_id]["away_allowed"]) / 10,
        }
        return team_features



    def get_team_stats(self, team_id: int):
        """
        Retrieve pre-computed team stats from team_map.
        :param team_id: The team identifier.
        :return: A dictionary of team stats.
        """
        if team_id not in self.team_map:
            raise ValueError(f"Team ID {team_id} not found in team_map.")
        return self.team_map[team_id]["stats"]

    def process_past_game(self, game: pd.Series, league_stats: dict):
        home_team_id = game["HID"]
        away_team_id = game["AID"]

        home_team_stats = self.get_team_stats(home_team_id)
        away_team_stats = self.get_team_stats(away_team_id)

        home_features = self.calculate_features_for_game(game, is_home_team=True, league_stats=league_stats, team_stats=home_team_stats)
        away_features = self.calculate_features_for_game(game, is_home_team=False, league_stats=league_stats, team_stats=away_team_stats)

        result = 1 if game["H"] == 1 else 0
        self.prediction_model.add_game_result(home_features, result)
        self.prediction_model.add_game_result(away_features, 1 - result)
  # Reverse result for the away team

    def predict_upcoming_game(self, game: pd.Series):
        """
        Predict the outcome of an upcoming game.
        :param game: Series containing game details.
        :return: Predicted probability of the home team winning.
        """
        home_team_id = game["HID"]
        away_team_id = game["AID"]

        # Define default league stats (example values, update as needed)
        league_stats = {
            "lg_ft": 47333,  # Total free throws made in the league
            "lg_pf": 49907,  # Total personal fouls in the league
            "lg_fta": 62008,  # Total free throw attempts in the league
        }

        # Calculate team-specific stats
        home_team_stats = self.get_team_stats(home_team_id)
        away_team_stats = self.get_team_stats(away_team_id)

        # Calculate features for the home team
        home_features = self.calculate_features_for_game(
            game, is_home_team=True, league_stats=league_stats, team_stats=home_team_stats
        )

        away_features = self.calculate_features_for_game(
            game, is_home_team=False, league_stats=league_stats, team_stats=away_team_stats
        )
        # dictionary of features
        # Home team 5 feature
        # Away team 5 feature
        # clash them
        # then we get the probability
        home_features
        away_features

        # Predict the probability of the home team winning
        return self.prediction_model.predict_outcome(home_features)

    def place_bets(self, summary: pd.DataFrame, opps: pd.DataFrame, inc: tuple[pd.DataFrame, pd.DataFrame]):
        """
        Make bets based on the current state of the environment.
        :param summary: Summary DataFrame with bankroll and betting limits.
        :param opps: Opps DataFrame with upcoming games and betting opportunities.
        :param inc: Tuple containing Games DataFrame (past games) and Players DataFrame (player statistics).
        :return: Bets DataFrame with suggested bets for home and away teams.
        """
        games_df = inc[0]
        players_df = inc[1]

        # Update stats with incremental data
        self.update_team_map(games_df)
        self.update_player_map(players_df)
        # self.update_leaderboards(players_df)

        # Define default league stats
        league_stats = {
            "lg_ft": 47333,  # Total free throws made in the league
            "lg_pf": 49907,  # Total personal fouls in the league
            "lg_fta": 62008,  # Total free throw attempts in the league
        }

        # Process past games
        for _, game in games_df.iterrows():
            self.process_past_game(game, league_stats=league_stats)

        # Train the prediction model
        if len(self.prediction_model.train_data) > 50:
            self.prediction_model.train_model()

        # Initialize bets DataFrame with zeros
        bets = pd.DataFrame(data=np.zeros((len(opps), 2)), columns=["BetH", "BetA"], index=opps.index)

        # Retrieve bankroll and betting limits
        bankroll = summary.iloc[0]["Bankroll"]
        max_bet = summary.iloc[0]["Max_bet"]
        min_bet = summary.iloc[0]["Min_bet"]  # Corrected to Min_bet

        self.iteration += 1
        # Iterate through betting opportunities
        for idx, opp in opps.iterrows():
            # Skip betting if the prediction model has insufficient training data
            if len(self.prediction_model.train_data) <= 500:
                continue

            # Predict probabilities for home and away teams
            predicted_prob_home = self.predict_upcoming_game(opp)
            predicted_prob_away = 1 - predicted_prob_home  # Complement of home team's probability

            odds_h = opp["OddsH"]
            odds_a = opp["OddsA"]

            # Calculate EV and place bet for the home team
            ev_home = predicted_prob_home * odds_h - (1 - predicted_prob_home)
            with open('output.txt', 'a') as f:
                f.write(f"predicted for home: {predicted_prob_home}" )
            if ev_home > 0 and predicted_prob_home > 0.85:
                bet_amount = min(bankroll * 0.08, max_bet)  # Cap at 5% of bankroll or max_bet
                bet_amount = max(bet_amount, min_bet)  # Ensure minimum bet limit
                bets.at[idx, "BetH"] = bet_amount  # Place bet on home team
                bankroll -= bet_amount  # Deduct from bankroll
        return bets


    